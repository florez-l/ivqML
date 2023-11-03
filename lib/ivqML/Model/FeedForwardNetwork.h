// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__FeedForwardNetwork__h__
#define __ivqML__Model__FeedForwardNetwork__h__

#include <vector>
#include <ivqML/Model/Base.h>

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
      using TConstMap = typename Superclass::TConstMap;

    public:
      FeedForwardNetwork( );
      virtual ~FeedForwardNetwork( ) override = default;

      virtual void set_number_of_parameters( const TNatural& p ) override;

      void add_layer(
        const TNatural& i, const TNatural& o, const std::string& a
        );
      void add_layer( const TNatural& o, const std::string& a );
      void init( );

      template< class _Y, class _X >
      void operator()(
        Eigen::EigenBase< _Y >& iY, const Eigen::EigenBase< _X >& iX,
        bool derivative = false
        ) const;

    protected:
      std::vector< TNatural > m_Sizes;
    };
  } // end namespace
} // end namespace

// TODO: #include <ivqML/Model/FeedForwardNetwork.hxx>

#endif // __ivqML__Model__FeedForwardNetwork__h__

// eof - $RCSfile$
