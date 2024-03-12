// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__NeuralNetwork__FeedForward__h__
#define __ivqML__Model__NeuralNetwork__FeedForward__h__

#include <cstdlib>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <ivqML/Model/Base.h>
#include <ivqML/Model/NeuralNetwork/ActivationFactory.h>

namespace ivqML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      /**
       */
      template< class _TScalar >
      class FeedForward
        : public ivqML::Model::Base< _TScalar >
      {
      public:
        using Self       = FeedForward;
        using Superclass = ivqML::Model::Base< _TScalar >;
        using TScalar    = typename Superclass::TScalar;
        using TNatural   = typename Superclass::TNatural;
        using TMat       = typename Superclass::TMat;
        using TCol       = typename Superclass::TCol;
        using TRow       = typename Superclass::TRow;
        using TMatMap    = typename Superclass::TMatMap;
        using TColMap    = typename Superclass::TColMap;
        using TRowMap    = typename Superclass::TRowMap;
        using TMatCMap   = typename Superclass::TMatCMap;
        using TColCMap   = typename Superclass::TColCMap;
        using TRowCMap   = typename Superclass::TRowCMap;

        using TActivationSignature = void( TMatMap&, const TMatMap&, bool );
        using TActivation = std::function< TActivationSignature >;
        using TActivationFactory =
          ivqML::Model::NeuralNetwork::ActivationFactory< Self >;

      public:
        FeedForward( );
        virtual ~FeedForward( ) override = default;

        virtual bool has_backpropagation( ) const override;

        virtual TNatural number_of_inputs( ) const override;
        virtual void set_number_of_inputs( const TNatural& p ) override;
        virtual TNatural number_of_outputs( ) const override;

        virtual TNatural buffer_size( ) const override;
        virtual void set_number_of_parameters( const TNatural& p ) final;

        void add_layer( const TNatural& i );
        void add_layer( const TNatural& o, const std::string& a );
        TNatural number_of_layers( ) const;
        void init( );

        TMatMap W( const TNatural& l );
        TMatCMap W( const TNatural& l ) const;

        TColMap B( const TNatural& l );
        TColCMap B( const TNatural& l ) const;

        template< class _TInputX >
        auto eval( const Eigen::EigenBase< _TInputX >& iX ) const;

        template< class _TInputX, class _TInputY >
        void backpropagation(
          TScalar* G,
          TScalar* B,
          const Eigen::EigenBase< _TInputX >& iX,
          const Eigen::EigenBase< _TInputY >& iY
          ) const;

        /* TODO
           template< class _TInputX >
           auto threshold( const Eigen::EigenBase< _TInputX >& iX ) const
           {
           return( iX.derived( ) );
           }

           template< class _TInputX, class _TInputY >
           void cost(
           TScalar* bG,
           const Eigen::EigenBase< _TInputX >& iX,
           const Eigen::EigenBase< _TInputY >& iY,
           TScalar* J = nullptr,
           TScalar* iB = nullptr
           ) const;
        */

      protected:
        template< class _TInputX >
        void _eval(
          const Eigen::EigenBase< _TInputX >& iX, TScalar* buffer
          ) const;

        virtual void _from_stream( std::istream& i ) override;
        virtual void _to_stream( std::ostream& o ) const override;

      protected:
        std::vector< TNatural > m_S;
        std::vector< TNatural > m_O;
        std::vector< std::pair< std::string, TActivation > > m_A;

        /* TODO
           bool m_IsLabeling { true }; // TODO: detect this
        */
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/Model/NeuralNetwork/FeedForward.hxx>

#endif // __ivqML__Model__NeuralNetwork__FeedForward__h__

// eof - $RCSfile$
