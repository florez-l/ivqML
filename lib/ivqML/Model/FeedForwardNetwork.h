// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__FeedForwardNetwork__h__
#define __ivqML__Model__FeedForwardNetwork__h__

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <ivqML/Model/Base.h>
#include <ivqML/Model/ActivationFactory.h>

namespace ivqML
{
  namespace Model
  {
    /**
     */
    template< class _S >
    class FeedForwardNetwork
      : public ivqML::Model::Base< _S >
    {
    public:
      using Self = FeedForwardNetwork;
      using Superclass = ivqML::Model::Base< _S >;

      using TScalar = typename Superclass::TScalar;
      using TNatural = typename Superclass::TNatural;
      using TMatrix = typename Superclass::TMatrix;
      using TMap = typename Superclass::TMap;

      using TSignature = void( TMap&, const TMap&, bool );
      using TActivation = std::function< TSignature >;
      using TActivationFactory = ivqML::Model::ActivationFactory< Self >;

    public:
      FeedForwardNetwork( );
      virtual ~FeedForwardNetwork( ) override = default;

      virtual void random_fill( ) override;

      virtual TNatural number_of_inputs( ) const override;
      virtual void set_number_of_inputs( const TNatural& p ) override;
      virtual TNatural number_of_outputs( ) const override;

      virtual void set_number_of_parameters( const TNatural& p ) final;

      void add_layer(
        const TNatural& i, const TNatural& o, const std::string& a
        );
      void add_layer( const TNatural& o, const std::string& a );
      TNatural number_of_layers( ) const;
      void init( );

    protected:
      virtual void _from_stream( std::istream& i ) override;
      virtual void _to_stream( std::ostream& o ) const override;

    protected:
      bool m_IsLabeling { true }; // TODO: detect this

      std::vector< TNatural > m_S;
      std::vector< TMap > m_W;
      std::vector< TMap > m_B;
      std::vector< std::pair< std::string, TActivation > > m_F;
    };
  } // end namespace
} // end namespace

#include <ivqML/Model/FeedForwardNetwork.hxx>

#endif // __ivqML__Model__FeedForwardNetwork__h__

// eof - $RCSfile$
